import React from 'react';
import { Card } from 'react-bootstrap';
import { MapContainer, TileLayer, Marker, Popup, useMap, Circle } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import $ from 'jquery';

// images
import iconGold from 'images/leaflet/marker-gold.png';
import iconModel from 'images/leaflet/marker-model.png';
import iconEntity from 'images/leaflet/marker-entity.png';
import iconGoldCCW from 'images/leaflet/marker-gold-ccw25.png';
import iconModelCW from 'images/leaflet/marker-model-cw25.png';

const EntityIcon = L.icon({
    iconUrl: iconEntity,
    iconSize: [40, 40],
    iconAnchor: [20, 45],
    popupAnchor: [0, -45]
});

const ModelIconCW = L.icon({
    iconUrl: iconModelCW,
    iconSize: [55, 55],
    iconAnchor: [17, 50],
    popupAnchor: [13, -42]
});

const GoldIconCCW = L.icon({
    iconUrl: iconGoldCCW,
    iconSize: [55, 55],
    iconAnchor: [35, 50],
    popupAnchor: [-10, -42]
});

const limeOptions = { color: '#18BC9C', opacity: 0.5 }

function ChangeView({center, radius}) {
    const map = useMap();
    map.closePopup();
    map.fitBounds(L.latLng(center).toBounds(radius));
    return null;
}

class Map extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: this.props.mapData,
            entities: [],
            modelCenter: [50.733334, 7.100000],
            trueCenter: {},
            zoom: 3,
            fixView: true,
            radius: 3500000,
            errorMessage: '',
            showPrediction: false
        };

        this.handleClick = this.handleClick.bind(this);
    }

    componentWillReceiveProps(nextProps) {
        this.setState(
            {
                data: nextProps.data,
                entities: nextProps.data.retrieved_entities,
                modelCenter: nextProps.data.pred_coords,
                trueCenter: nextProps.data.true_coords,
                fixView: nextProps.fixMapView,
                radius: nextProps.data.query.radius * 1000,
                showPrediction: JSON.stringify(nextProps.data) === '{}' ? false : true
            }
        );
    }

    handleClick (event) {
        var data = {
            id: event.target.options.id,
            label: this.state.data.retrieved_entities.filter(obj => {return obj.id === event.target.options.id})[0].label
        };
        $("#overlay").fadeIn(300);
        axios.post('/api/select_image_news_events', data)
            .then(response => {
                $("#overlay").fadeOut(300);
                this.props.appCallback(response.data);
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
                $("#overlay").fadeOut(300);
            	console.error('There was an error while requesting results for the selected image!', error);
            });
    }

    render () {
        return (
            <>
                <Card className="border-light mb-3">
                    <MapContainer scrollWheelZoom={true} style={{height: "500px", width: "100%"}}>
                        {this.state.fixView ? <ChangeView center={this.state.modelCenter} radius={this.state.radius} /> : <></>}
                        <TileLayer
                            attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                        {this.state.entities.map((entity, index) => (
                            <Marker
                                key={entity.id}
                                id={entity.id}
                                position={entity.coordinates}
                                eventHandlers={{click: (event) => {this.handleClick(event)}}}
                                icon={EntityIcon}>
                                <Popup>
                                    {entity.label}
                                </Popup>
                            </Marker>
                        ))}
                        {this.state.showPrediction ? <Circle center={this.state.modelCenter} pathOptions={limeOptions} radius={this.state.radius} /> : <></>}
                        {this.state.showPrediction ?
                            <Marker
                                position={this.state.modelCenter}
                                icon={ModelIconCW}
                                zIndexOffset={1000}
                                rotationAngle={45}>
                                <Popup>
                                    Model prediction: {this.state.modelCenter.lat}, {this.state.modelCenter.lng}
                                </Popup>
                            </Marker> : <></>}
                        {this.state.showPrediction && JSON.stringify(this.state.trueCenter) !== '{}' ?
                            <Marker
                                position={this.state.trueCenter}
                                icon={GoldIconCCW}
                                zIndexOffset={1000}>
                                <Popup>
                                    Ground truth: {this.state.trueCenter.lat}, {this.state.trueCenter.lng}
                                </Popup>
                            </Marker> : <></>}
                        </MapContainer>
                    <Card.Footer style={{textAlign: "center"}}>
                        <small className="text-muted" style={{marginRight: "10px"}}>Marker:</small>
                        <img src={iconModel} alt={'model marker'} width={'24em'} height={'24em'} />
                        <small className="text-muted" style={{marginRight: "10px"}}>Model</small>
                        <img src={iconGold} alt={'ground truth marker'} width={'24em'} height={'24em'} />
                        <small className="text-muted" style={{marginRight: "10px"}}>Ground truth</small>
                        <img src={iconEntity} alt={'entity marker'} width={'24em'} height={'24em'} />
                        <small className="text-muted">Entity</small>
                    </Card.Footer>
                </Card>
            </>
        );
    }
}

export default Map;
