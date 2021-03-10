# GeoWINE: Geolocation based Wiki, Image, News and Events Retrieval

In the context of social media, geolocation inference on news or events has become a very important task. In this paper, we present the GeoWINE (Geolocation-based Wiki-Image-News-Event retrieval) demonstrator, an effective modular system for multimodal retrieval which expects only a single image as input. The GeoWINE system consists of five modules in order to retrieve related information from various sources. The first module is a state-of-the-art model for geolocation estimation of images. The second module performs a geospatial-based query for entity retrieval using the Wikidata knowledge graph. The third module exploits four different image embedding representations, which are used to retrieve most similar entities compared to the input image. The embeddings are derived from the tasks of geolocation estimation, place recognition, ImageNet-based image classification, and their combination. The last two modules perform news and event retrieval from EventRegistry and the Open Event Knowledge Graph (OEKG). GeoWINE provides an intuitive interface for end-users and is insightful for experts for reconfiguration to individual setups. The GeoWINE achieves promising results in entity label prediction for images on Google Landmarks dataset. The demonstrator is publicly available at [cleopatra.ijs.si/geowine/](http://cleopatra.ijs.si/geowine/).

![GeoWINE](src/images/architecture.png?raw=true "CARTON architecture")

Overview of the GeoWINE architecture.

## Requirements
### Frontend
The frontend was built using React JS (JavaScript), an open-source JavaScript library for user interface components.
The following packages are arrequired:
- [Node.js](https://nodejs.org/en/)
- [Yarn](https://yarnpkg.com/)

### Backend
Our backend is served as a Python Flask application. The following libraries are required:

- Python version >= 3.7
- PyTorch version = 1.5.1

``` bash
# clone the repository
git clone https://github.com/cleopatra-itn/GeoWINE

cd GeoWINE/api/

# create a virtual environment
python3 -m venv venv
source venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## Run GeoWINE
### Frontend
You can start the frontend using the yarn package:

``` bash
# start frontend
yarn start
```

On [package.json](package.json), we have configured the proxy address for redirecting all requests to the backend.

### Backend
Similarly,  we can run our backend using the yarn package:

``` bash
# start backend
yarn start-api
```

In order to run the backend properly, you will need to cache the entity embeddings. To do so, please run all the files on the [scripts](api/scripts) folder.

## License
The repository is under [MIT License](LICENSE).

## Cite
Coming Soon!