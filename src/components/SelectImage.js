import React from 'react';
import { Button } from 'react-bootstrap';
import ImageGallery from 'react-image-gallery';
import axios from 'axios';
import "react-image-gallery/styles/css/image-gallery.css";

const IMAGES = [
    {
      original: require('images/resized/Q172251.jpg').default,
      thumbnail: require('images/thumbnail/Q172251.jpg').default,
      id: 'Q172251'
    },
    {
      original: require('images/resized/Q675042.jpg').default,
      thumbnail: require('images/thumbnail/Q675042.jpg').default,
      id: 'Q675042'
    },
    {
      original: require('images/resized/Q808535.jpg').default,
      thumbnail: require('images/thumbnail/Q808535.jpg').default,
      id: 'Q808535'
    },
    {
      original: require('images/resized/Q988217.jpg').default,
      thumbnail: require('images/thumbnail/Q988217.jpg').default,
      id: 'Q988217'
    },
    {
      original: require('images/resized/Q1012650.jpg').default,
      thumbnail: require('images/thumbnail/Q1012650.jpg').default,
      id: 'Q1012650'
    },
    {
      original: require('images/resized/Q2625123.jpg').default,
      thumbnail: require('images/thumbnail/Q2625123.jpg').default,
      id: 'Q2625123'
    },
    {
      original: require('images/resized/Q6704480.jpg').default,
      thumbnail: require('images/thumbnail/Q6704480.jpg').default,
      id: 'Q6704480'
    },
    {
      original: require('images/resized/Q7476191.jpg').default,
      thumbnail: require('images/thumbnail/Q7476191.jpg').default,
      id: 'Q7476191'
    },
    {
      original: require('images/resized/Q11726080.jpg').default,
      thumbnail: require('images/thumbnail/Q11726080.jpg').default,
      id: 'Q11726080'
    },
    {
      original: require('images/resized/Q15077594.jpg').default,
      thumbnail: require('images/thumbnail/Q15077594.jpg').default,
      id: 'Q15077594'
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
        axios.post('/select', data) // submit to api and get results
            .then(response => {
				this.props.inputImageCallback(response.data); // Pass response data to parent
				console.log(response.data)
			})
            .catch(error => {
            	this.setState({ errorMessage: error.message });
            	console.error('There was an error!', error);
        });
    }

    render () {
        return (
            <div>
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
            </div>
        );
    }
}

export default SelectImage;
