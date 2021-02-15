import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import InfoModal from 'components/InfoModal';
import ReferenceModal from 'components/ReferenceModal';
import logo from 'logo.svg';

function Header() {
    return (
        <Navbar bg="primary" expand="lg" className='App-navbar'>
            <Navbar.Brand href="#">
                <img
                    alt=""
                    src={logo}
                    width="30"
                    height="30"
                    className="d-inline-block align-top"
                />
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
                <Nav className="mr-auto">
                    <InfoModal />
                    <ReferenceModal />
                    <Nav.Link href="https://github.com/cleopatra-itn/geolocation-demo.git" target="_blank">GitHub</Nav.Link>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    );
}

export default Header;
