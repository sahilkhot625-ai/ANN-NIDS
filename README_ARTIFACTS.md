# Save these from your training notebook into the `artifacts/` folder

```python
import os, json, joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs("artifacts", exist_ok=True)
model.save("artifacts/ann_model.keras")
joblib.dump(scaler, "artifacts/scaler.pkl")
json.dump(["Normal", "Attack"], open("artifacts/classes.json", "w"))   # or your multiclass names

# y_test = true labels, y_pred = predicted labels (0/1 or class index)
json.dump({
  "accuracy":  accuracy_score(y_test, y_pred),
  "precision": precision_score(y_test, y_pred, average="weighted"),
  "recall":    recall_score(y_test, y_pred, average="weighted"),
  "f1":        f1_score(y_test, y_pred, average="weighted"),
}, open("artifacts/metrics.json", "w"))
np.save("artifacts/confusion_matrix.npy", confusion_matrix(y_test, y_pred))
pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.to_csv("artifacts/classification_report.csv")
pd.DataFrame(history.history).to_csv("artifacts/history.csv", index=False)
```
