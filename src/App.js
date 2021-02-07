import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import Header from 'components/Header';
import Footer from 'components/Footer';
import InputImageTab from 'components/InputImageTab';
import Map from 'components/Map';
import ResultsTab from 'components/ResultsTab';
import 'App.css';

class App extends React.Component {

    state = {
        responseData: {},
        selectedEntity: '',
        resultData: {
            entity: 'Entities',
            newsArticles: 'News articles',
            events: 'Events'
        }
    }

    callbackResponse = (newResponseData) => {
        this.setState(
            {
                responseData: newResponseData
            }
        );
        console.log('We got the grand child response data!!!');
    }

    callbackSelected = (selectedEntityOnMap) => {
        this.setState(
            {
                selectedEntity: selectedEntityOnMap,
                resultData: {
                    entity: 'Entities from response',
                    newsArticles: 'News articles from response',
                    events: 'Events from response'
                }
            }
        );
        console.log('Selected entity came to parent!!!');
    }

    render () {
        return (
        <>
            <Header />

            <Container fluid>
                <Row>
                    <Col sm={4}>
                        <InputImageTab
                            appCallback={this.callbackResponse}
                        />
                    </Col>

                    <Col sm={5}>
                        <Map
                            appCallback={this.callbackSelected}
                        />
                    </Col>

                    <Col sm={3}>
                        <ResultsTab
                            resultDataFromApp={this.state.resultData}
                        />
                    </Col>
                </Row>
            </Container>

            <Footer />
        </>
        );
    }
}

export default App;
