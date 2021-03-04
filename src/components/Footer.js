import React from 'react'
import { Container } from 'react-bootstrap';
import './../App.css'

class Footer extends React.Component {
    render () {
        return (
            <footer className="App-footer">
                <Container fluid>
                    <span>&copy; Copyright 2021: GeoWINE - v{0.1}</span><br/>
                    <span className='App-footer-references'>Built with <a href="https://getbootstrap.com/" target="_blanck">Bootstrap</a>. Theme from <a href="https://bootswatch.com/" target="_blanck">Bootswatch</a>.</span>
                </Container>
            </footer>
        );
    }
}

export default Footer;
