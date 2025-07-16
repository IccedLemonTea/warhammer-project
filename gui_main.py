import sys

import os
os.environ["DISPLAY"] = ":0"
os.environ["QT_QPA_PLATFORM"] = "xcb"

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms'


from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QComboBox,
    QVBoxLayout, QHBoxLayout, QSpacerItem, QSizePolicy,
    QPushButton, QStackedLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage


## Causing QT problems
import depthai as dai
import cv2
from ultralytics import YOLO


class WarhammerDiceCheckerUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Warhammer Dice Checker")
        self.setGeometry(100, 100, 600, 400)

        self.setStyleSheet("""
            QWidget {
                background-color: #0f1a2b; /* Deep dark blue */
                color: #e0e0e0;           /* Light grey text */
                font-family: 'Eurostile', 'Segoe UI', sans-serif;
                font-size: 14px;
            }

            QLabel {
                color: #f8f8f8; /* White-ish for headers */
                font-weight: bold;
            }

            QComboBox {
                background-color: #1e2c3a; /* Navy-grey hybrid */
                color: #ffffff;
                border: 1px solid #445566;
                padding: 6px;
                border-radius: 3px;
            }

            QComboBox QAbstractItemView {
                background-color: #2a3b4d;
                color: #ffffff;
                selection-background-color: #a00c0c;  /* Blood red highlight */
            }

            QPushButton {
                background-color: #293845;
                color: #ffffff;
                border: 1px solid #4c5a66;
                padding: 6px 10px;
                border-radius: 4px;
            }

            QPushButton:hover {
                background-color: #3b4c5c;
                border: 1px solid #ff3b3b; /* red glow on hover */
            }

            QPushButton:pressed {
                background-color: #a00c0c; /* red pulse */
                border: 1px solid #ffaaaa;
            }

            QLineEdit, QTextEdit {
                background-color: #1e2c3a;
                color: #ffffff;
                border: 1px solid #445566;
            }

            QScrollBar:vertical {
                background: #1a1f25;
                width: 10px;
            }

            QScrollBar::handle:vertical {
                background: #445566;
                border-radius: 4px;
            }
        """)

        self.stack = QStackedLayout()
        self.setLayout(self.stack)


        self.army_logos = {"Space Marines": "", "Black Templars": "", "Blood Angels": "", "Dark Angels": "", "Grey Knights": "", "Space Wolves": "", "Death Watch": "",
                                    "Adepta Sororitas": "", "Adeptus Custodes": "", "Adeptus Mechanicus": "", "Astra Militarum": "",
                                    "Chaos Space Marines": "", "Death Guard": "", "Thousand Sons": "", "World Eaters": "army images/ZerkerTrim.png", "Emperor's Children": "",
                                    "Aeldari": "", "Drukhari": "", "Orkz": "", "Tau": "", "Tyranids": "", "Genestealer Cults": "", "Leagues of Votann": "", "Necrons": ""}
        # Build both screens
        self.army_page = self.init_army_choice()
        self.combat_page = self.init_combat_choice()
        

        self.video_label = QLabel()  # Will show video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.detection_page = self.init_dice_detection()

        self.pipeline = None
        self.device = None
        self.queue = None

        self.stack.addWidget(self.army_page)   # Index 0
        self.stack.addWidget(self.combat_page) # Index 1
        self.stack.addWidget(self.detection_page) # Index 2

        self.stack.setCurrentIndex(0)

    def init_army_choice(self):
        page = QWidget()
        main_layout = QHBoxLayout()

        # Attacker layout
        attacker_layout = QVBoxLayout()
        attacker_label = QLabel("Attacker Faction:")
        self.attacker_dropdown = QComboBox()
        self.attacker_dropdown.addItems(list(self.army_logos.keys()))
        self.attacker_dropdown.currentTextChanged.connect(self.update_attacker_logo)

        # Logo QLabel
        self.attacker_logo = QLabel()


        attacker_layout.addWidget(attacker_label)
        attacker_layout.addWidget(self.attacker_dropdown)
        attacker_layout.addWidget(self.attacker_logo)

        attacker_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Defender layout
        defender_layout = QVBoxLayout()
        defender_label = QLabel("Defender Faction:")
        self.defender_dropdown = QComboBox()
        self.defender_dropdown.addItems(list(self.army_logos.keys()))
        defender_layout.addWidget(defender_label)
        defender_layout.addWidget(self.defender_dropdown)
        defender_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        save_continue = QPushButton("Save && Continue")

        # Change to combat screen
        save_continue.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        defender_layout.addWidget(save_continue)

        # Combine both sides
        main_layout.addLayout(attacker_layout)
        main_layout.addLayout(defender_layout)
        page.setLayout(main_layout)
        return page

    def init_combat_choice(self):
        page = QWidget()
        main_layout = QHBoxLayout()

        # Attacker layout -- Need to add query system based on attacker army choice, and weapon characterstic per model
        attacker_layout = QVBoxLayout()
        attacker_model_label = QLabel("Attacker Unit:")
        attacker_model_dropdown = QComboBox()
        attacker_model_dropdown.addItems(["Khorne Berzerker", "Exalted Eightbound", "Jackhals", "Angron", "World Eaters Terminators"])
        attacker_weapon_label = QLabel("Weapon Profile:")
        attacker_weapon_dropdown = QComboBox()
        attacker_weapon_dropdown.addItems(["Bolt Pistol", "Chainsword", "Khornate Eviscerator"])

        attacker_layout.addWidget(attacker_model_label)
        attacker_layout.addWidget(attacker_model_dropdown)
        attacker_layout.addItem(QSpacerItem(0, 40, QSizePolicy.Minimum, QSizePolicy.Fixed))
        attacker_layout.addWidget(attacker_weapon_label)
        attacker_layout.addWidget(attacker_weapon_dropdown)
        attacker_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Defender layout -- Need to add query system based on defender army choice, and weapon characterstic per model
        defender_layout = QVBoxLayout()
        defender_model_label = QLabel("Defender Unit:")
        defender_model_dropdown = QComboBox()
        defender_model_dropdown.addItems(["Khorne Berzerker", "Exalted Eightbound", "Jackhals", "Angron", "World Eaters Terminators"])
        defender_weapon_label = QLabel("Model Characterstic:")
        defender_weapon_dropdown = QComboBox()
        defender_weapon_dropdown.addItems(["Bolt Pistol", "Chainsword", "Khornate Eviscerator"])

        defender_layout.addWidget(defender_model_label)
        defender_layout.addWidget(defender_model_dropdown)
        defender_layout.addItem(QSpacerItem(0, 40, QSizePolicy.Minimum, QSizePolicy.Fixed))
        defender_layout.addWidget(defender_weapon_label)
        defender_layout.addWidget(defender_weapon_dropdown)
        defender_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        save_continue = QPushButton("Save && Continue")
        save_continue.clicked.connect(lambda: self.stack.setCurrentIndex(2))
        defender_layout.addWidget(save_continue)

        main_layout.addLayout(attacker_layout)
        main_layout.addLayout(defender_layout)
        page.setLayout(main_layout)
        return page

    def init_dice_detection(self):
        page = QWidget()
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Dice Detection Mode:"))
        layout.addWidget(self.video_label)

        start_btn = QPushButton("Start Camera")
        start_btn.clicked.connect(self.start_pipeline)
        layout.addWidget(start_btn)

        page.setLayout(layout)
        return page

    def start_pipeline(self):
        # Basic mono camera + color preview pipeline
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(640, 480)
        cam.setInterleaved(False)
        cam.setFps(30)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("video")
        cam.preview.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)

        self.timer.start(30)  # refresh ~33 fps

    def update_frame(self):
        if self.queue:
            in_frame = self.queue.get()
            frame = in_frame.getCvFrame()
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_attacker_logo(self, army_name):
        if army_name in self.army_logos:
            pixmap = QPixmap(self.army_logos[army_name])
            self.attacker_logo.setPixmap(pixmap)
        else:
            self.attacker_logo.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WarhammerDiceCheckerUI()
    window.show()
    sys.exit(app.exec_())
