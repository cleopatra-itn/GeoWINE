import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';
import Select, { components } from 'react-select';
import ImageGallery from 'react-image-gallery';
import axios from 'axios';
import RangeSlider from 'react-bootstrap-range-slider';
import $ from 'jquery';

const IMAGES = [
    {
      original: require('images/input/notreparis.jpg').default,
      thumbnail: require('images/input/notreparis.jpg').default,
      id: 'notreparis.jpg'
    },
    {
        original: require('images/input/noterdam.jpeg').default,
        thumbnail: require('images/input/noterdam.jpeg').default,
        id: 'noterdam.jpeg'
    },
    {
        original: require('images/input/timessquare.jpeg').default,
        thumbnail: require('images/input/timessquare.jpeg').default,
        id: 'timessquare.jpeg'
    },
    {
        original: require('images/input/vatican.jpeg').default,
        thumbnail: require('images/input/vatican.jpeg').default,
        id: 'vatican.jpeg'
    }
];

const optionsTypes = [
    { value: 'tourist attraction', label: 'Tourist Attraction' },
    { value: 'church', label: 'Church' },
    { value: 'skyscraper', label: 'Skyscraper' },
    { value: 'tower', label: 'Tower' },
    { value: 'building', label: 'Building' },
    { value: 'monument', label: 'Monument' },
    { value: 'historic', label: 'Historic' },
    { value: 'bridge', label: 'Bridge' },
    { value: 'museum', label: 'Museum' },
    { value: 'square', label: 'Square' },
    { value: 'castle', label: 'Castle' },
    { value: 'waterfall', label: 'Waterfall' }
];

const ValueContainer = ({ children, getValue, ...props }) => {
    var length = getValue().length;

    return (
      <components.ValueContainer {...props}>
        {!props.selectProps.menuIsOpen &&
          `${length} selected`}
        {React.cloneElement(children[1])}
      </components.ValueContainer>
    );
  };

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
            selectedTypeOption: [optionsTypes[0]],
            selectedRadiusOption: 10
        };

        this.handleClick = this.handleClick.bind(this);
        this.handleTypeChange = this.handleTypeChange.bind(this);
      }

    handleClick () {
        if (this.state.selectedTypeOption.length === 0) {
            alert('Please select at least one type.');
            return 0;
        }

        // POST data and get results
        var data = {
            id: IMAGES[this.state.activeIndex]['id'],
            type: this.state.selectedTypeOption,
            radius: this.state.selectedRadiusOption,
        };
        $("#overlay").fadeIn(300);
        axios.post('/api/select_image_entities', data) // submit to api and get results
            .then(response => {
                $("#overlay").fadeOut(300);
                this.props.inputImageCallback(response.data); // pass response data to parent
            })
            .catch(error => {
                this.setState({ errorMessage: error.message });
                $("#overlay").fadeOut(300);
                alert(error.message);
                console.error('There was an error while requesting results for the selected image!', error);
            });
    }

    handleTypeChange (selectedTypeOption) {
        this.setState({ selectedTypeOption });
    };

    render () {
        return (
            <>
                <Card className={'border-light mb-3'} style={{height: '510px', zIndex: '1000'}}>
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


                    <Row style={{marginTop: '0.5rem'}}>
                        <Col xs={5}>
                            <label>Type of results:</label>
                        </Col>

                        <Col xs={7} style={{zIndex: '1000'}}>
                            <Select
                                value={this.state.selectedTypeOption}
                                onChange={this.handleTypeChange}
                                options={optionsTypes}
                                isMulti
                                className="basic-multi-select"
                                classNamePrefix="select"
                                name="colors"
                                components={{ ValueContainer }}
                                hideSelectedOptions={false}
                                closeMenuOnSelect={false}
                                isClearable={false}
                            />
                        </Col>
                    </Row>

                    <Row style={{marginTop: '1.8rem'}}>
                        <Col xs={5}>
                            <label>Radius of results:</label>
                        </Col>

                        <Col xs={7} style={{zIndex: '100'}}>
                            <RangeSlider
                                value={this.state.selectedRadiusOption}
                                onChange={changeEvent => this.setState({ selectedRadiusOption: changeEvent.target.value })}
                                min='1'
                                max='25'
                                tooltipPlacement='top'
                                tooltip='on'
                                className="custom-range"
                            />
                        </Col>
                    </Row>

                    <Button onClick={this.handleClick} variant="primary" size="lg" block style={{marginTop: '1rem'}}>
                        Predict location &amp; Get Results
                    </Button>
                    </Card.Body>
                </Card>
            </>
        );
    }
}

export default SelectImage;
