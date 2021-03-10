import React from 'react';
import ImageUploader from "react-images-upload";
import { Button, Card } from 'react-bootstrap';
import { Row, Col } from 'react-bootstrap';
import Select from 'react-select';
import axios from 'axios';
import RangeSlider from 'react-bootstrap-range-slider';
import $ from 'jquery';
import { TYPES, ValueContainer } from 'components/Constants';

class UploadImage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            imageFIle: [],
            previewImage: '',
            selectedTypeOption: [TYPES[0]],
            selectedRadiusOption: 10,
            errorMessage: '',
        };

        this.handleClick = this.handleClick.bind(this);
        this.handleTypeChange = this.handleTypeChange.bind(this);
        this.onDrop = this.onDrop.bind(this);
    }

    handleClick () {
        if (this.state.imageFIle.length === 0) {
            alert('Please choose an image.');
            return 0;
        }

        if (this.state.selectedTypeOption.length === 0) {
            alert('You need to select at least one type.');
            return 0;
        }

        // POST data and get results
        var form_data = new FormData();
        form_data.append('file', this.state.imageFIle[0]);
        form_data.append('type', JSON.stringify(this.state.selectedTypeOption));
        form_data.append('radius', JSON.stringify(this.state.selectedRadiusOption));

        $("#overlay").fadeIn(300);
        axios.post('/api/upload_image_entities', form_data)
            .then(response => {
                $("#overlay").fadeOut(300);
                this.props.inputImageCallback(response.data); // pass response data to parent
            })
            .catch(error => {
                this.setState({ errorMessage: error.message });
                $("#overlay").fadeOut(300);
                alert(error.message);
                console.error('There was an error while requesting results for the uploaded image!', error);
            });
    }

    onDrop (pictureFile, pictureDataURLs) {
        console.log(pictureFile)
        this.setState({
          imageFIle: pictureFile,
          previewImage: URL.createObjectURL(pictureFile[0])
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
                        <ImageUploader
                            withIcon={false}
                            buttonText="Choose images"
                            onChange={this.onDrop}
                            imgExtension={[".jpg", ".jpeg", ".png"]}
                            maxFileSize={20971520}
                            label={'Max file size: 20mb, accepted: jpg|jpeg|png'}
                            singleImage={true}
                        />

                        <div style={{height: '200px', maxWidth: '100%'}}>
                            {this.state.previewImage && (
                                <img style={{maxHeight: '100%', maxWidth: '100%'}} src={this.state.previewImage} alt="" />
                            )}
                        </div>


                        <Row style={{marginTop: '0.5rem'}}>
                            <Col xs={5}>
                                <label>Type of results:</label>
                            </Col>

                            <Col xs={7} style={{zIndex: '1000'}}>
                                <Select
                                    value={this.state.selectedTypeOption}
                                    onChange={this.handleTypeChange}
                                    options={TYPES}
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

export default UploadImage;