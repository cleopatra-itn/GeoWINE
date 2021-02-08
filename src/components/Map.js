import React, { useEffect } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import icon from 'images/leaflet/marker-icon.png';
import iconShadow from 'images/leaflet/marker-shadow.png';


let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 40],
    iconAnchor: [13, 45],
    popupAnchor: [0, -45],
});

class Map extends React.Component {
    render () {
        return (
            <>
                <MapContainer className="jumbotron" center={[51.505, -0.09]} zoom={13} scrollWheelZoom={false} style={{height: "500px", width: "100%"}}>
                    <TileLayer
                        attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
                        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />
                    <Marker
                        position={[51.505, -0.09]}
                        eventHandlers={{click: () => {this.props.appCallback('Entity selected')}}}
                        icon={DefaultIcon}>
                        <Popup>
                            A pretty CSS3 popup. <br /> Easily customizable.
                        </Popup>
                    </Marker>
                </MapContainer>
            </>
        );
    }
}

export default Map;
