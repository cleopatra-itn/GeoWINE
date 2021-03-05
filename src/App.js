import React from 'react';
import { Container, Row, Col } from 'react-bootstrap';
import Footer from 'components/Footer';
import InputImageTab from 'components/InputImageTab';
import Map from 'components/Map';
import ResultsTab from 'components/ResultsTab';
import OrderedResults from  'components/OrderedResults';
import LoaderOverlay from  'components/LoaderOverlay';

class App extends React.Component {

    state = {
        data: {},
        fixMapView: true,
        retrievedEntities: [],
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
                retrievedEntities: newResponseData.retrieved_entities,
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
            <Container fluid style={{marginTop: '20px'}}>
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

                <Row>
                    <Col>
                        <OrderedResults
                            dataFromApp={this.state.retrievedEntities}
                        />
                    </Col>
                </Row>
            </Container>

            <LoaderOverlay />

            <Footer />
        </>
        );
    }
}

export default App;
