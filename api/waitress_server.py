import waitress, api
waitress.serve(api.app, host='0.0.0.0', port=9990)