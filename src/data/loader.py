import pandas as pd
from pathlib import Path

def load_data(path):
    columns =["square_id",
          "time_interval",
          "country_code",
          "sms_in",
          "sms_out",
          "call_in",
          "call_out",
          "internet"]
    data = pd.read_csv(path,
                       sep="\t",
                       header=None,
                       names=columns
                      )


    data["time_interval"] = pd.to_datetime(data["time_interval"],unit='ms')
    data["time_interval"] = data["time_interval"].dt.tz_localize("UTC").dt.tz_convert('Europe/Rome')
    data = data.fillna(0)
    data = data.groupby(["square_id","time_interval"]).sum()
    data = data.drop(columns=["country_code"])
    data = data.reset_index()


    return data



def load_total_data(num_days, path, verbose=True):
    path = Path(path)
    files = sorted(path.glob("sms-call-internet-mi-*.txt"))[:num_days]
    
    data_list = []
    for i, f in enumerate(files, start=1):
        try:
            data_list.append(load_data(f))
            if verbose:
                bar = "#" * i + "=" * (num_days - i)
                print(f"[{bar}] {i*100//num_days}%", end="\r")
        except Exception as e:
            print(f"failed at {f.name}: {e}")
    
    if not data_list:
        raise FileNotFoundError(f"{path} içinde hiç veri dosyası bulunamadı")
    
    return pd.concat(data_list, ignore_index=True)