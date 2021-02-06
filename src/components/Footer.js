import React from 'react'
import { Container } from 'react-bootstrap';
import './../App.css'

class Footer extends React.Component {
    render () {
        return (
            <footer className="App-footer">
                <Container fluid>
                    <span>&copy; Copyright 2021: Geolocation-Demo - v{0.1}</span><br/>
                    <span className='App-footer-references'>Based on <a href="https://getbootstrap.com/" target="_blanck">Bootstrap</a>. Theme from <a href="https://bootswatch.com/" target="_blanck">Bootswatch</a>. Icons from <a href="https://fontawesome.com/" target="_blanck">Font Awesome</a>. Web fonts from <a href="https://fonts.google.com/" target="_blanck">Google</a>.</span>
                </Container>
            </footer>
        );
    }
}

export default Footer;
