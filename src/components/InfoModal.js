import React, { useState } from 'react';
import { Nav, Modal } from 'react-bootstrap';

function InfoModal() {
    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);

    return (
      <>
        <Nav.Link href="#" onClick={handleShow}>Info</Nav.Link>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Info</Modal.Title>
          </Modal.Header>
          <Modal.Body>Info Text</Modal.Body>
        </Modal>
      </>
    );
  }

export default InfoModal;