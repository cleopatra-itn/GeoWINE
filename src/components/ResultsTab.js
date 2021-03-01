import React from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import Entities from 'components/Entities';
import NewsArticles from 'components/NewsArticles';
import Events from 'components/Events';

class ResultsTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            key: 'entityTab',
            id: this.props.dataFromApp.id,
            entity: this.props.dataFromApp.entity,
            news: this.props.dataFromApp.news,
            events: this.props.dataFromApp.events,
        };
    }

    componentWillReceiveProps(nextProps) {
        this.setState(
            {
                key: JSON.stringify(nextProps.dataFromApp.entity) === '{}' ? 'entityTab' : this.state.key,
                id: nextProps.dataFromApp.id,
                entity: nextProps.dataFromApp.entity,
                news: nextProps.dataFromApp.news,
                events: nextProps.dataFromApp.events,
            }
        );
    }

    render () {
        return (
            <Tabs
                id="controlled-tab-example"
                activeKey={this.state.key}
                onSelect={(k) => this.setState({ key: k })}
                >
                <Tab eventKey="entityTab" title="Entity">
                    <Entities
                        data={this.state.entity}
                    />
                </Tab>
                {JSON.stringify(this.state.news) === '[]' ?
                    <Tab eventKey='disabled' title="News Articles" disabled>
                    </Tab> :
                    <Tab eventKey='newsArticlesTab' title="News Articles">
                        <NewsArticles
                            data={this.state.news}
                        />
                    </Tab>
                }
                {JSON.stringify(this.state.events) === '[]' ?
                    <Tab eventKey='disabled' title="Events" disabled>
                    </Tab> :
                    <Tab eventKey="EventsTab" title="Events">
                        <Events
                            data={this.state.events}
                        />
                    </Tab>
                }
            </Tabs>
        );
    }
}

export default ResultsTab;
