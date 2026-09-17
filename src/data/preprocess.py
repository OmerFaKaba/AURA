import pandas as pd

def select_valid_cells(data, min_total=1000, min_std=10, verbose=True):
    """
    Modellenebilir hücreleri seçer.
    Kriterler: toplam aktivite eşiği + varyans eşiği.
    Dönüş: (gecerli_id_listesi, elenen_id_listesi)
    """
    total = data.groupby("square_id")["internet"].sum()
    std = data.groupby("square_id")["internet"].std()

    mask = (total >= min_total) & (std >= min_std)

    valid = total[mask].index.tolist()
    dropped = total[~mask].index.tolist()

    if verbose:
        print(f"toplam hücre : {len(total)}")
        print(f"geçerli      : {len(valid)}")
        print(f"elenen       : {len(dropped)}")
        if dropped:
            print(f"elenen id'ler: {dropped[:10]}{' ...' if len(dropped) > 10 else ''}")

    return valid, dropped



def to_hourly_pivot(df, value_col="internet", freq_raw="10min", verbose=True):
    """Uzun formatlı veriyi saatlik geniş formata çevirir.

    Adımlar: pivot -> eksik zaman dilimlerini doldur -> saatlik topla.
    Dönüş: satırlar zaman, sütunlar square_id olan DataFrame.
    """
    pivot = df.pivot_table(
        index="time_interval",
        columns="square_id",
        values=value_col,
        aggfunc="sum",
    )

    full_axis = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq_raw)
    missing = len(full_axis) - len(pivot)

    pivot = pivot.reindex(full_axis).fillna(0)
    hourly = pivot.resample("1h").sum()

    if verbose:
        print(f"pivot        : {pivot.shape}")
        print(f"doldurulan   : {missing} zaman dilimi")
        print(f"saatlik      : {hourly.shape}")

    return hourly