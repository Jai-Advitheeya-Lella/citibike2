import hopsworks
import mlflow
import joblib
import os

# ✅ Connect to Hopsworks
project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project="citybike",
    api_key_value="t4rwmi4VfZBIaFqR.ulzGrQ0eIDKCUKgaPYEpNH4JHWuXbIu3YU8gogK8ldP5tpUpnMVIG4uQX1BFW9Wb"
)
mr = project.get_model_registry()

# ✅ Load best MLflow model
mlflow.set_experiment("citibike_trip_prediction_v2")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("citibike_trip_prediction_v2")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mae ASC"])

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/best_tuned_model"

# ✅ Load the model and save it locally
model = mlflow.sklearn.load_model(model_uri)
model_dir = "model_artifacts"
os.makedirs(model_dir, exist_ok=True)
model_file_path = os.path.join(model_dir, "best_lgbm_model.pkl")
joblib.dump(model, model_file_path)

# ✅ Register model using the correct method
model_hops = mr.python.create_model(
    name="citibike_best_model",
    metrics={"mae": best_run.data.metrics["mae"]},
    description="Best tuned LightGBM model with lag and time features"
)

model_hops.save(model_dir)  # ✅ This is the correct usage
print(f"✅ Best model registered to Hopsworks with MAE: {best_run.data.metrics['mae']:.2f}")
