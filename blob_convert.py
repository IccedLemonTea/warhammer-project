import blobconverter

blob_path = blobconverter.from_onnx(
    model='/home/warhammer/Downloads/best.onnx',
    data_type='FP16',        # Smaller + fast
    shaves=6,                # Number of SHAVEs on OAK-1
    use_cache=False,
    optimizer_params=["--mean_values=[0,0,0]", "--scale_values=[255,255,255]"]
)

print(f"Blob saved to: {blob_path}")


## API tapi.NOIhb0gonafgcDeGjTiYAQ.qZK11TywmSMG5PNzJi7gSyDDTwXkJDO_Jyu0S0sJF2hHbhj82AjI_3MunySuTxycZ6F9IOKGol32vuXuq7yf_A