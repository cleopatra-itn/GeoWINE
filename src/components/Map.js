import React from 'react';
import { Card } from 'react-bootstrap';
import { MapContainer, TileLayer, Marker, Popup, useMap, Circle } from 'react-leaflet';
import L from 'leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import iconEntity from 'images/leaflet/marker-entity.png';
import iconModel from 'images/leaflet/marker-model.png';
import iconGold from 'images/leaflet/marker-gold.png';
// import iconShadow from 'images/leaflet/marker-shadow.png';

const EntityIcon = L.icon({
    iconUrl: iconEntity,
    // shadowUrl: iconShadow,
    iconSize: [40, 40],
    iconAnchor: [20, 45],
    popupAnchor: [0, -45]
});

const ModelIcon = L.icon({
    iconUrl: iconModel,
    // shadowUrl: iconShadow,
    iconSize: [40, 40],
    iconAnchor: [20, 45],
    popupAnchor: [0, -45]
});

const GoldIcon = L.icon({
    iconUrl: iconGold,
    // shadowUrl: iconShadow,
    iconSize: [40, 40],
    iconAnchor: [20, 45],
    popupAnchor: [0, -45]
});

const limeOptions = { color: '#18BC9C', opacity: 0.2 }

function ChangeView({center, zoom, radius}) {
    const map = useMap();
    // const marker = L.marker(center)
    // var latLngs = [ marker.getLatLng() ];
    // const circle = L.circle(center, radius);
    map.fitBounds(L.latLng(center).toBounds(radius));
    // const marker = L.marker(center)
    // var latLngs = [ marker.getLatLng() ];
    // var markerBounds = L.latLngBounds(latLngs);
    // map.fitBounds(markerBounds);
    // map.setView(center, zoom);
    return null;
}

class Map extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: this.props.mapData,
            entities: [],
            modelCenter: [50.733334, 7.100000],
            trueCenter: [50.733334, 7.100000],
            zoom: 3,
            fixView: true,
            radius: 3500000,
            errorMessage: ''
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
                radius: nextProps.data.query.radius * 1000
            }
        );
    }

    handleClick (event) {
        var data = { id: event.target.options.id };
        axios.post('/select_image_news_events', data)
            .then(response => {
                this.props.appCallback(response.data);
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
            	console.error('There was an error while requesting results for the selected image!', error);
            });
    }

    render () {
        return (
            <>
                <Card className="border-light mb-3">
                    <MapContainer scrollWheelZoom={true} style={{height: "500px", width: "100%"}}>
                        {this.state.fixView ? <ChangeView center={this.state.modelCenter} zoom={this.state.zoom} radius={this.state.radius} /> : <></>}
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
                        {this.state.entities.length > 0 ? <Circle center={this.state.modelCenter} pathOptions={limeOptions} radius={this.state.radius} /> : <></>}
                        {this.state.entities.length > 0 ?
                            <Marker
                                position={this.state.modelCenter}
                                icon={ModelIcon}>
                                <Popup>
                                    Model prediction: {this.state.modelCenter[0]}, {this.state.modelCenter[1]}
                                </Popup>
                            </Marker> : <></>}
                            {this.state.entities.length > 0 ?
                            <Marker
                                position={this.state.trueCenter}
                                icon={GoldIcon}>
                                <Popup>
                                    Ground truth: {this.state.trueCenter[0]}, {this.state.trueCenter[1]}
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
