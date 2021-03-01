import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';
import Select from 'react-select';
import ImageGallery from 'react-image-gallery';
import axios from 'axios';
import $ from 'jquery';
import "react-image-gallery/styles/css/image-gallery.css";

const IMAGES = [
    {
      original: require('images/resized/Q2981.jpg').default,
      thumbnail: require('images/thumbnail/Q2981.jpg').default,
      id: 'notreparis.jpg' // 'Q2981'
    }
];

const optionsTypes = [
    { value: 'Q570116', label: 'Tourist Attraction' }, // 2870
    { value: 'Q33506', label: 'Museum' }, // 40031
    { value: 'Q839954', label: 'Archaeological Site' }, // 39211
    { value: 'Q618123', label: 'Geographical Feature' }, // 6604
    { value: 'Q2319498', label: 'Landmark' }, // 627
    { value: 'Q43229', label: 'Organization' }, // 70283
    { value: 'Q327333', label: 'Government Agency' }, // 12729
    { value: 'Q1802963', label: 'Mansion' }, // 1323
    { value: 'Q162875', label: 'Mausoleum' }, // 2061
    { value: 'Q2221906', label: 'Geographic Location' }, // 8300
    { value: 'Q2065736', label: 'Cultural Property' }, // 66003
    { value: 'Q41176', label: 'Building' } //
];

const optionsRadius = [
    { value: '1', label: 'Street Level (1KM)' },
    { value: '25', label: 'City Level (25KM)' },
    { value: '200', label: 'Region Level (200KM)' },
    { value: '750', label: 'Country Level (750KM)' }
];

class SelectImage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            items: IMAGES,
            showNav: false,
            autoPlay: false,
            showPlayButton: false,
            showFullscreenButton: false,
            useBrowserFullscreen: false,
			activeIndex: 0,
			errorMessage: '',
            selectedTypeOption: { value: 'Q570116', label: 'Tourist Attraction' },
            selectedRadiusOption: { value: '25', label: 'City Level (25KM)' }
        };

        this.handleClick = this.handleClick.bind(this);
        this.handleTypeChange = this.handleTypeChange.bind(this);
        this.handleRadiusChange = this.handleRadiusChange.bind(this);
      }

    handleClick () {
        // POST data and get results
        console.log('Sending')
        var data = {
            id: IMAGES[this.state.activeIndex]['id'],
            type: this.state.selectedTypeOption.value,
            radius: this.state.selectedRadiusOption.value,
        };
        $("#overlay").fadeIn(300);
        axios.post('/api/select_image_entities', data) // submit to api and get results
            .then(response => {
                console.log(response.data);
                $("#overlay").fadeOut(300);
                this.props.inputImageCallback(response.data); // pass response data to parent
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
                $("#overlay").fadeOut(300);
            	console.error('There was an error while requesting results for the selected image!', error);
            });
    }

    handleTypeChange (selectedTypeOption) {
        this.setState({ selectedTypeOption });
    };

    handleRadiusChange (selectedRadiusOption) {
        this.setState({ selectedRadiusOption });
    };

    render () {
        return (
            <>
                <Card className={'border-light mb-3'} style={{height: '510px'}}>
                    <Card.Body style={{padding: '0.5rem'}}>
                    <ImageGallery
                        items={this.state.items}
                        showNav={this.state.showNav}
                        autoPlay={this.state.autoPlay}
                        showPlayButton={this.state.showPlayButton}
                        showFullscreenButton={this.state.showFullscreenButton}
                        useBrowserFullscreen={this.state.useBrowserFullscreen}
                        onThumbnailClick={(_, index ) => this.setState({ activeIndex: index })}
                    />


                    <Row style={{marginTop: '1rem'}}>
                        <Col xs={3}>
                            <label>Type:</label>
                        </Col>

                        <Col xs={9}>
                            <Select
                                value={this.state.selectedTypeOption}
                                onChange={this.handleTypeChange}
                                options={optionsTypes}
                            />
                        </Col>
                    </Row>

                    <Row style={{marginTop: '0.5rem'}}>
                        <Col xs={3}>
                            <label>Radius:</label>
                        </Col>

                        <Col xs={9}>
                            <Select
                                value={this.state.selectedRadiusOption}
                                onChange={this.handleRadiusChange}
                                options={optionsRadius}
                            />
                        </Col>
                    </Row>

                    <Button onClick={this.handleClick} variant="primary" size="lg" block style={{marginTop: '1rem'}}>
                        Guess Location &amp; Get Results
                    </Button>
                    </Card.Body>
                </Card>
            </>
        );
    }
}

export default SelectImage;
