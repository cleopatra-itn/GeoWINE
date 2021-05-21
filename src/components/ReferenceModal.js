import React, { useState } from 'react';
import { Nav, Modal } from 'react-bootstrap';

function ReferenceModal() {
    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);

    return (
      <>
        <Nav.Link style={{display: 'inline', padding: '0.5rem'}} href="#" onClick={handleShow}>Reference</Nav.Link>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Reference</Modal.Title>
          </Modal.Header>
          <Modal.Body>
            Golsa Tahmasebzadeh, Endri Kacupaj, Eric Müller-Budack, Sherzod Hakimov, Jens Lehmann, and Ralph Ewerth. 2021. GeoWINE: Geolocation based Wiki, Image, News and Event Retrieval. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, Online
          </Modal.Body>
        </Modal>
      </>
    );
  }

export default ReferenceModal;