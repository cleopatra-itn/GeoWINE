import React from 'react';
import { Button } from 'react-bootstrap';
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
]

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
			errorMessage: ''
        };

        this.handleClick = this.handleClick.bind(this);
      }

    handleClick () {
        // POST data and get results
        var data = { id: IMAGES[this.state.activeIndex]['id'] };
        axios.post('/select_image_entities', data) // submit to api and get results
            .then(response => {
                this.props.inputImageCallback(response.data); // pass response data to parent
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
            	console.error('There was an error while requesting results for the selected image!', error);
            });
    }

    render () {
        return (
            <>
                <ImageGallery
                    items={this.state.items}
                    showNav={this.state.showNav}
                    autoPlay={this.state.autoPlay}
                    showPlayButton={this.state.showPlayButton}
                    showFullscreenButton={this.state.showFullscreenButton}
                    useBrowserFullscreen={this.state.useBrowserFullscreen}
                    onThumbnailClick={(_, index ) => this.setState({ activeIndex: index })}
                />
                <Button onClick={this.handleClick} variant="primary" size="lg" block>
                    Guess Location &amp; Get Results
                </Button>
            </>
        );
    }
}

export default SelectImage;
