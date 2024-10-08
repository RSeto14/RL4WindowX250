import json

# JSONへの保存
def dataclass_to_json(dataclass_instance,file_path="config.json"):
    json_data = json.dumps(dataclass_instance, default=lambda o: o.__dict__, indent=4)
    # ファイルに保存
    with open(file_path, 'w') as f:
        f.write(json_data)

# JSONの読み込み
def json_to_dataclass(file_path="config.json"):
    with open(file_path, 'r') as f:
        loaded_json_data = f.read()
    data = json.loads(loaded_json_data)
    return data


if __name__ == "__main__":
    #　使用例
    
    from Scripts.cfg import Config
    # configの作成
    config = Config()
    print(config)
    config.gpu = 7
    config.seed = 1111
    
    # JSONへの保存
    dataclass_to_json(config,file_path="config.json")
    
    # JSONからの読み込み
    data = json_to_dataclass(file_path="config.json")
    loaded_config = Config(**data)
    print("JSONから読み込まれたデータ:")
    print(loaded_config)
    
    