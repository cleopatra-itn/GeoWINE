import React, { useState } from 'react';
import { Nav, Modal } from 'react-bootstrap';

function InfoModal() {
    const [show, setShow] = useState(false);

    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);

    return (
      <>
        <Nav.Link style={{display: 'inline', padding: '0.5rem'}} href="#" onClick={handleShow}>Info</Nav.Link>
        <Modal show={show} onHide={handleClose}>
          <Modal.Header closeButton>
            <Modal.Title>Info</Modal.Title>
          </Modal.Header>
          <Modal.Body style={{height: '500px', overflowY: 'auto'}}>
              <p>
                GeoWINE (Geolocation-based Wiki-Image-News-Event retrieval) is a multimodal retrieval system based on five main modules: Geolocation estimation, geospatial-based query for entity retrieval, image representation, news retrieval and event retrieval.
              </p>
              <p>
                This system could be useful in many areas such as: fact-checking, fake news detection and image verification.
              </p>
              <p>
                Inputs expected from the user are: an image (either selected from the samples or uploaded by the user), radius in kilometers and entity types.
                Given the inputs, the geolocation estimation module pinpoints the predicted location on the map. Next based on the predicted location you can see other outputs provided by the system:
              </p>
              <ul>
                <li>The entity retrieval module retrieves entities from Wikidata within the selected radius from the predicted location and presents them on the map.</li>
                <li>The image representation module sorts the retrieved entities based on similarities to the input image and shows them on the bottom.</li>
                <li>If you are interested in the current news or events taking place in any of the retrieved entities, just click the entity on the map and enjoy the news!</li>
              </ul>
              <p>
              GeoWINE provides the current news and events taking place in any of the retrieved entities using EventRegistry and EventKg respectively.
              The source code for the system is <a href="https://github.com/cleopatra-itn/GeoWINE" target="_blank" rel="noreferrer">publicly available</a>.
              </p>
          </Modal.Body>
        </Modal>
      </>
    );
  }

export default InfoModal;