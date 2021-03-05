import React from 'react'
import { Nav } from 'react-bootstrap';
import { Container } from 'react-bootstrap';
import InfoModal from 'components/InfoModal';
import ReferenceModal from 'components/ReferenceModal';

class Footer extends React.Component {
    render () {
        return (
            <footer className="App-footer">
                <Container fluid>
                    <div style={{marginBottom: '1rem'}}>
                        <InfoModal /> |
                        <ReferenceModal /> |
                        <Nav.Link style={{display: 'inline', padding: '0.5rem'}} href="https://github.com/cleopatra-itn/geolocation-demo.git" target="_blank">GitHub</Nav.Link>
                    </div>
                    <span>&copy; Copyright 2021: GeoWINE - v{0.1}</span><br/>
                    <span className='App-footer-references'>Built with <a href="https://getbootstrap.com/" target="_blanck">Bootstrap</a>. Theme from <a href="https://bootswatch.com/" target="_blanck">Bootswatch</a>.</span>
                </Container>
            </footer>
        );
    }
}

export default Footer;
