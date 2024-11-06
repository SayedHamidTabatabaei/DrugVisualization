import os

from presentation import create_app
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
