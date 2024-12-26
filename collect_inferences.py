import json,re
from pathlib import Path
import pandas as pd

def collect_inf(chkpoint_num:int=120, target_collect_path:str='P:\\我的雲端硬碟\\models\\predictions'):
    # Variables to format into the filename pattern
    var1 = '.+'  # Replace with your actual variable
    var2 = chkpoint_num  # Replace with your actual variable

    # Define the target path and filename pattern
    target_collect_path = Path(target_collect_path)
    pattern = re.compile(fr'qid{var1}checkpoint{var2}.json')
    # Find all files matching the pattern
    all_files = list(target_collect_path.iterdir())
    # Exclude "ALL.json" and collect matching files
    filtered_files = [f for f in all_files if pattern.match(f.name) and f.name.find('pred_retrieve')==-1]
    if len(filtered_files)<=0:
        return None
    # List to hold all records
    all_records = []
    # Load JSON content from each file and collect the records
    for file in filtered_files:
        with file.open('r', encoding='utf-8') as json_file:
            datadict = json.load(json_file)
        records = pd.DataFrame.from_records(datadict)
        all_records.append(records)

    # Concatenate all dataframes and reset the index
    concatenated_df = pd.concat(all_records) \
        .drop_duplicates() \
        .reset_index(drop=True)
    # Group the concatenated dataframe by columns "qid", "category", "retrieve"
    grouped_df = concatenated_df.groupby(['qid', 'category', 'retrieve'])[['qid','category','retrieve','incre_num_by']]
    # Select the row with the smallest "incre_num_by" within each group
    filtered_df = grouped_df.apply(lambda x: x.loc[x['incre_num_by'].idxmin()], include_groups=False).reset_index(drop=True)
    filtered_df['qid'] = filtered_df['qid'].astype(int)
    filtered_df['retrieve'] = filtered_df['retrieve'].astype(int)
    filtered_df = filtered_df.drop(columns=['incre_num_by','category']) \
        .sort_values(by=['qid']) \
        .reset_index(drop=True)
    # Convert the final dataframe to a list of records (dicts)
    all_records = filtered_df.to_dict(orient='records')

    # Write the collected records to "ALL.json"
    all_json_path = target_collect_path.parent / f'pred_retrieve_{chkpoint_num}.json'
    all_records = {"answers":all_records}
    with all_json_path.open('w', encoding='utf-8') as json_file:
        json.dump(all_records, json_file, ensure_ascii=False, indent=4)

    print(f"Collected records have been written to {all_json_path}")

if __name__=='__main__':
    import argparse,time
    parser = argparse.ArgumentParser(
        description="Example script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument('--chkpoint_num', type=int, default=120, help="checkpoint number")
    parser.add_argument('--target_collect_path', type=str, default='P:\\我的雲端硬碟\\models\\predictions', help="target collect path")
    parser.add_argument('--sleep_seconds', type=int, default=20, help="sleep seconds")
    args = parser.parse_args()
    argp_dict = vars(args)
    while True:
        collect_inf(chkpoint_num=120)
        collect_inf(chkpoint_num=480)
        collect_inf(chkpoint_num=600)
        time.sleep(argp_dict['sleep_seconds'])