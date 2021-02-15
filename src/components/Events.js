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
                {this.state.data.map((event, _) => (
                    <Card className="border-light mb-3">
                        <Card.Body>
                            <Card.Title>{event.title}</Card.Title>
                            <Card.Subtitle className="mb-2 text-muted">{event.date}</Card.Subtitle>
                            <Card.Text className={'cut-text'}>
                                {event.description}
                            </Card.Text>
                            <Card.Link href='http://oekg.l3s.uni-hannover.de/' target='_blank'>Read more</Card.Link>
                        </Card.Body>
                    </Card>
                ))}
            </Card>
        );
    }
}

export default Events;
