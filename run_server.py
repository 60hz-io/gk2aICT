import json
import os

from flask import (
    Flask,
    request,
    jsonify
)


app = Flask(__name__)


@app.route('/')
def index():
    return '<h2>60hz satelite data API server</h2>'


@app.route('/data', methods=['GET'])
def get_data():
    """ Returns json type data requested by client
    
    Parameters will be given in json(or dictionary)

    Parameters
    ----------
    target_data: string
        name of target data

    target_datetime: string
        Datetime in string format 'YYYYMMDDhhmm'

    Returns
    -------
    result: json
        Requested data must be decoded (by client)
    """
    # get arguments(parameters) from request
    data_name = request.args.get('target_data')
    target_datetime = request.args.get('target_datetime')
    # type check of target_datetime 
    if type(target_datetime) == int:
        target_datetime = str(target_datetime)
    elif type(target_datetime) != str:
        target_datetime = target_datetime.strftime("%Y%m%d%H%M")
    target_datetime = target_datetime.replace('-', '')

    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        data_name
    )
    file_name = f'gk2a_{data_name}_{target_datetime}.json'

    target = load_data(os.path.join(file_path, file_name))

    return jsonify(result)


def load_data(file_path):
    """ Returns data in requested file_path
    
    If data is not exist, rasie Exception
    
    Parameters
    ----------
    file_path: string or os.path
        Path where target_data exist
        
    Returns
    ------
    data: json
        File is saved as json type, but raw data is of verious types.
        If raw data is np.array, 
        you should decode it using base64 as in 'example_request_data.ipynb'
    """

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e

    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
