# test_depthai.py
import depthai as dai

try:
    with dai.Device() as device:
        print("OAK device connected!")
        print("USB speed:", device.getUsbSpeed())
except Exception as e:
    print("Error connecting to OAK device:")
    print(e)
