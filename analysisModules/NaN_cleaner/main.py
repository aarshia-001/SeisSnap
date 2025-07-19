MODULE_NAME = "NaN Cleaner"
DESCRIPTION = "Removes NaNs from the dataset."

def configure(parent=None):
    from PySide6.QtWidgets import QInputDialog
    method, ok = QInputDialog.getItem(parent, "Cleaning Method", "Select Method:", ["dropna", "fillna"], 0, False)
    if not ok:
        return {}
    return {"method": method}

def run(df, settings):
    if settings.get("method") == "dropna":
        return df.dropna()
    elif settings.get("method") == "fillna":
        return df.fillna(0)
    return df
