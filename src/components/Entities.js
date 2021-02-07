import React from 'react';

class Entities extends React.Component {
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
            <p>Entities: {this.state.data}</p>
        );
    }
}

export default Entities;
