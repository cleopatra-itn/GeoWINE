import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';
import Select from 'react-select';
import ImageGallery from 'react-image-gallery';
import axios from 'axios';
import "react-image-gallery/styles/css/image-gallery.css";

const IMAGES = [
    {
      original: require('images/resized/Q2981.jpg').default,
      thumbnail: require('images/thumbnail/Q2981.jpg').default,
      id: 'Q2981'
    },
    {
      original: require('images/resized/Q82878.jpg').default,
      thumbnail: require('images/thumbnail/Q82878.jpg').default,
      id: 'Q82878'
    }
];

const optionsTypes = [
    { value: 'Q570116', label: 'Tourist Attraction' },
    { value: 'Q33506', label: 'Museum' },
];

const optionsRadius = [
    { value: '5', label: '5KM' },
    { value: '10', label: '10KM' },
    { value: '20', label: '20KM' },
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
            selectedRadiusOption: { value: '5000', label: '5KM' }
        };

        this.handleClick = this.handleClick.bind(this);
        this.handleTypeChange = this.handleTypeChange.bind(this);
        this.handleRadiusChange = this.handleRadiusChange.bind(this);
      }

    handleClick () {
        // POST data and get results
        var data = {
            id: IMAGES[this.state.activeIndex]['id'],
            type: this.state.selectedTypeOption.value,
            radius: this.state.selectedRadiusOption.value,
        };
        axios.post('/select_image_entities', data) // submit to api and get results
            .then(response => {
                this.props.inputImageCallback(response.data); // pass response data to parent
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
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
                        <Col sm={3}>
                            <label>Type:</label>
                        </Col>

                        <Col sm={9}>
                            <Select
                                value={this.state.selectedTypeOption}
                                onChange={this.handleTypeChange}
                                options={optionsTypes}
                            />
                        </Col>
                    </Row>

                    <Row style={{marginTop: '0.5rem'}}>
                        <Col sm={3}>
                            <label>Radius:</label>
                        </Col>

                        <Col sm={9}>
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
