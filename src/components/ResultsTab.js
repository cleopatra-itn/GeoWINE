import React from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import Entities from 'components/Entities';
import NewsArticles from 'components/NewsArticles';
import Events from 'components/Events';

class ResultsTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            key: 'entityTab'
          };
      }

    render () {
        return (
            <Tabs
                id="controlled-tab-example"
                activeKey={this.key}
                onSelect={(k) => this.setState({ key: k })}
                >
                <Tab eventKey="entityTab" title="Entity">
                    <Entities />
                </Tab>
                <Tab eventKey="newsArticlesTab" title="News Articles">
                    <NewsArticles />
                </Tab>
                <Tab eventKey="EventsTab" title="Events">
                    <Events />
                </Tab>
            </Tabs>
        );
    }
}

export default ResultsTab;
