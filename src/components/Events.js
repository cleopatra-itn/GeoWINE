import React from 'react';
import { Card } from 'react-bootstrap';

class Events extends React.Component {
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
            <Card className={'border-light mb-3'} style={{height: "510px", overflowY: "auto", padding: '1.25rem'}}>
                {this.state.data.sort((a, b) => (a.date > b.date) ? -1 : 1).map((event, _) => (
                    <Card className="border-light mb-3">
                        <Card.Body>
                            <Card.Title>{event.label}</Card.Title>
                            <Card.Subtitle className="mb-2 text-muted">{event.date}</Card.Subtitle>
                            <Card.Text className={'cut-text'}>
                                {event.en_description}
                            </Card.Text>
                            <Card.Link href={event.oekg_uri} target='_blank'>Read more</Card.Link>
                        </Card.Body>
                    </Card>
                ))}
            </Card>
        );
    }
}

export default Events;
