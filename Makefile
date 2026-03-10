

setup : 
	pip install -e . 

launch-mlflow-server :
	echo "Launching MLflow server... "
	echo "All parameters are default in this command but you can customize them as needed."
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 0.0.0.0 --port 5000