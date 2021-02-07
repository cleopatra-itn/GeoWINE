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
            entityData: this.props.resultDataFromApp.entity,
            newsArticlesData: this.props.resultDataFromApp.newsArticles,
            eventsData: this.props.resultDataFromApp.events,
        };
    }

    componentWillReceiveProps(nextProps) {
        this.setState(
            {
                entityData: nextProps.resultDataFromApp.entity,
                newsArticlesData: nextProps.resultDataFromApp.newsArticles,
                eventsData: nextProps.resultDataFromApp.events,
            }
        );
    }

    render () {
        return (
            <Tabs
                id="controlled-tab-example"
                activeKey={this.key}
                onSelect={(k) => this.setState({ key: k })}
                >
                <Tab eventKey="entityTab" title="Entity">
                    <Entities
                        data={this.state.entityData}
                    />
                </Tab>
                <Tab eventKey="newsArticlesTab" title="News Articles">
                    <NewsArticles
                        data={this.state.newsArticlesData}
                    />
                </Tab>
                <Tab eventKey="EventsTab" title="Events">
                    <Events
                        data={this.state.eventsData}
                    />
                </Tab>
            </Tabs>
        );
    }
}

export default ResultsTab;
