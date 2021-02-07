import React from 'react';

class NewsArticles extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: this.props.data
        };
    }

    componentWillReceiveProps(nextProps) {
        this.setState(
            {
                data: nextProps.data
            }
        );
    }

    render () {
        return (
            <p>News Articles: {this.state.data}</p>
        );
    }
}

export default NewsArticles;