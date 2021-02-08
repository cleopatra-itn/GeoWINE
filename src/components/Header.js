import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import InfoModal from 'components/InfoModal';
import ReferenceModal from 'components/ReferenceModal';
import logo from 'logo.svg';

function Header() {
    return (
        <Navbar bg="light" expand="lg" className='App-navbar'>
            <Navbar.Brand href="#home">
                <img
                    alt=""
                    src={logo}
                    width="30"
                    height="30"
                    className="d-inline-block align-top"
                />{' '}
                Geolocation Demo
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="mr-auto">
                    <InfoModal />
                    <ReferenceModal />
                </Nav>
            </Navbar.Collapse>
            <Nav.Link href="https://github.com/cleopatra-itn/geolocation-demo.git" target="_blank">GitHub</Nav.Link>
        </Navbar>
    );
}

export default Header;
