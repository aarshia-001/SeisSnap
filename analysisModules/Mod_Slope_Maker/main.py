MODULE_NAME = "Mod_Slope_Maker"
DESCRIPTION = "Creates slope velocity column based on two selected columns and offset."

def configure(parent=None):
    from PySide6.QtWidgets import QInputDialog

    # Get offset as float input
    offset, ok = QInputDialog.getDouble(parent, "Offset", "Enter offset multiplier:", 1.0, decimals=4)
    if not ok:
        return {}

    # Get col1 (time-like)
    col1, ok = QInputDialog.getText(parent, "Time Column", "Enter name of time-like column (col1):")
    if not ok or not col1.strip():
        return {}

    # Get col2 (distance-like)
    col2, ok = QInputDialog.getText(parent, "Distance Column", "Enter name of distance-like column (col2):")
    if not ok or not col2.strip():
        return {}

    return {"offset": offset, "col1": col1.strip(), "col2": col2.strip()}


def run(df, settings):
    import pandas as pd

    offset = settings.get("offset")
    col1 = settings.get("col1")
    col2 = settings.get("col2")

    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"One or both columns '{col1}' and '{col2}' not found in DataFrame.")

    # Drop rows with NaNs in col1 or col2
    df = df.dropna(subset=[col1, col2])

    # Drop duplicates in col1
    df = df.drop_duplicates(subset=col1)

    # Sort by col1 ascending
    df = df.sort_values(by=col1).reset_index(drop=True)

    # Compute slope velocity
    slope = []
    for i in range(len(df) - 1):
        delta_time = df.loc[i + 1, col1] - df.loc[i, col1]
        delta_dist = df.loc[i + 1, col2] - df.loc[i, col2]
        if delta_time == 0:
            slope.append(None)
        else:
            slope.append(offset * (delta_dist / delta_time))

    # Last entry has no i+1 to compare to
    slope.append(None)

    df["SlopeVelocity"] = slope

    # Drop rows where SlopeVelocity is NaN
    df = df.dropna(subset=["SlopeVelocity"]).reset_index(drop=True)

    return df
