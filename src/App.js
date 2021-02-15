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
        data: {},
        fixMapView: true,
        selectedEntity: {
            id: '',
            entity: {},
            news: [],
            events: [],
        }
    }

    callbackResponse = (newResponseData) => {
        this.setState(
            {
                data: newResponseData,
                fixMapView: true,
                selectedEntity: {
                    id: '',
                    entity: {},
                    news: [],
                    events: [],
                },
            }
        );
    }

    callbackResponseEntity = (newResponseData) => {
        this.setState(
            {
                selectedEntity: {
                    id: newResponseData.id,
                    entity: this.state.data.retrieved_entities.filter(obj => {return obj.id === newResponseData.id})[0],
                    news: newResponseData.news,
                    events: newResponseData.events,
                },
                fixMapView: false
            }
        );
    }

    render () {
        return (
        <>
            <Header />

            <Container fluid>
                <Row>
                    <Col sm={3}>
                        <InputImageTab
                            appCallback={this.callbackResponse}
                        />
                    </Col>

                    <Col sm={5}>
                        <Map
                            data={this.state.data}
                            fixMapView={this.state.fixMapView}
                            appCallback={this.callbackResponseEntity}
                        />
                    </Col>

                    <Col sm={4}>
                        <ResultsTab
                            dataFromApp={this.state.selectedEntity}
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
