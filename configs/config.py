import json
import os


def load_config(env):
    config_file = f'configs/config_{env}.json'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            conf = json.load(f)
        return conf
    else:
        raise FileNotFoundError(f'Config file {config_file} not found')


os_env = os.environ.get('ENV', 'dev')
config = load_config(os_env)

mysql_host = config['mysql']['host']
mysql_user = config['mysql']['user']
mysql_password = config['mysql']['password']
mysql_database_name = config['mysql']['database']

batch_size = int(config['management']['batch_size'])
enable_bert_embedding = int(config['management']['enable_bert_embedding'])
