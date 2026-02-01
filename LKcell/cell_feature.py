import os
import pandas as pd
import ipdb
# ipdb.set_trace()

paths = pd.read_csv('/home/stat-huamenglei/LKCell/slide_ov_response.csv')['slides_name'].tolist()
for path in paths:
    data = os.listdir(path)
    csv_paths = []
    csv_paths.extend([os.path.join(path, d) for d in data if d.endswith('.csv')])
    valid_csv_paths = [p for p in csv_paths if os.path.getsize(p) > 0]  # 只保留非空
    print("有效CSV文件数：", len(valid_csv_paths))
    dfs = []
    for csv_file in valid_csv_paths:
        # print(csv_file)
        ipdb.set_trace()
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"⚠️ 跳过空文件: {csv_file}")
    # ipdb.set_trace()
    all_df = pd.concat(dfs, ignore_index=True)

    cols = [
        'major_axis_length',
        'minor_axis_length',
        'major_minor_ratio',
        'orientation_degree',
        'area',
        'extent',
        'solidity',
        'convex_area',
        'eccentricity',
        'equivalent_diameter',
        'perimeter'
    ]

    result = all_df.groupby('cell_type')[cols].mean().reset_index()
    save_dir = os.path.dirname(path)  
    save_path_mean = os.path.join(save_dir, "cell_feature_mean.csv")
    save_path_all = os.path.join(save_dir, "cell_feature_all.csv")
    print(result)
    all_df.to_csv(save_path_all, index=False)
    result.to_csv(save_path_mean, index=False)