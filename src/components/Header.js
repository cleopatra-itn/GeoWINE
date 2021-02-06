import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import logo from 'logo.svg';

class Header extends React.Component {
    render () {
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
                        <Nav.Link href="#">Info</Nav.Link>
                        <Nav.Link href="#">GitHub</Nav.Link>
                        <Nav.Link href="#">Reference</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Navbar>
        );
    }
}

export default Header;
