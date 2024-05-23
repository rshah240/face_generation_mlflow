from mlflow import MlflowClient


client = MlflowClient()
client.create_registered_model("dcgan_face")
result = client.create_model_version(
    name = "dcgan_face",
    source = "mlartifacts/167417142991466618/f8d4efee73f740d59e6da3d16e3e9c0d/artifacts/model",
    run_id = "f8d4efee73f740d59e6da3d16e3e9c0d"
)

client.set_registered_model_alias("dcgan_face", "best_model", 1)