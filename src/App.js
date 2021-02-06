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
        responseData: {}
    }

    callbackResponse = (newResponseData) => {
        this.setState(
            {
                responseData: newResponseData
            }
        );
        console.log('We got the grand child response data!!!')
    }

    render () {
        return (
        <div>
            <Header />

            <Container fluid>
                <Row>
                    <Col sm={4}>
                        <InputImageTab appCallback = {this.callbackResponse} />
                    </Col>

                    <Col sm={5}>
                        <Map />
                    </Col>

                    <Col sm={3}>
                        <ResultsTab />
                    </Col>
                </Row>
            </Container>

            <Footer />
        </div>
        );
    }
}

export default App;
