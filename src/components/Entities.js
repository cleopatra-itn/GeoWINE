import React from 'react';
import { Card, ListGroup, ListGroupItem } from 'react-bootstrap';

class Entities extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: this.props.data
        };
    }

    componentWillReceiveProps(nextProps) {
        this.setState(
            {
                data: nextProps.data
            }
        );
    }

    render () {
        return (
            <>
            <Card className={'border-light mb-3'} style={{height: '510px', overflowY: 'auto'}}>
                {JSON.stringify(this.state.data) === '{}' ?
                    <Card.Body>
                        <Card.Title>No entity selected.</Card.Title>
                    </Card.Body> :
                <div>
                    <div className={'wrapper'}>
                        <Card.Img variant="top" src={this.state.data.image_url} />
                    </div>
                    <Card.Body>
                        <Card.Title>{this.state.data.label}</Card.Title>
                        <Card.Subtitle className="mb-2 text-muted">{this.state.data.id} ({Object.values(this.state.data.types)})</Card.Subtitle>
                        <Card.Text className={'cut-text'}>
                            {this.state.data.en_description}
                        </Card.Text>
                    </Card.Body>
                    <ListGroup className="list-group-flush">
                        <ListGroupItem>Geolocation similarity: {this.state.data.similarity_geolocation}</ListGroupItem>
                        <ListGroupItem>Scene similarity: {this.state.data.similarity_scene}</ListGroupItem>
                        <ListGroupItem>Object similarity: {this.state.data.similarity_obj}</ListGroupItem>
                        <ListGroupItem>Total similarity: {this.state.data.similarity_all}</ListGroupItem>
                    </ListGroup>
                    <Card.Body>
                        <Card.Link href={this.state.data.entity_uri} target='_blank'>Wikidata</Card.Link>
                        <Card.Link href={this.state.data.wikipedia_page} target='_blank'>Wikipedia</Card.Link>
                        <Card.Link href='https://commons.wikimedia.org/wiki/Main_Page' target='_blank'>Wikimedia Commons</Card.Link>
                    </Card.Body>
                </div>
                }
            </Card>
            </>
        );
    }
}

export default Entities;
