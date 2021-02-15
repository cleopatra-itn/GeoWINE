import React from 'react';
import { Card } from 'react-bootstrap';

class NewsArticles extends React.Component {
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
                {this.state.data.map((article, _) => (
                    <Card className="border-light mb-3">
                        <Card.Body>
                            <Card.Title>{article.title}</Card.Title>
                            <div style={{display: 'flex', flexDirection: 'row', flexWrap: 'nowrap', justifyContent: 'space-between'}}>
                                <Card.Subtitle className="mb-2 text-muted">Source: {article.source}</Card.Subtitle>
                                <Card.Subtitle className="mb-2 text-muted">{article.date}</Card.Subtitle>
                            </div>
                            <Card.Text className={'cut-text'}>
                                {article.body}
                            </Card.Text>
                            <Card.Link href={article.url} target='_blank'>Read more</Card.Link>
                        </Card.Body>
                    </Card>
                ))}
            </Card>
        );
    }
}

export default NewsArticles;