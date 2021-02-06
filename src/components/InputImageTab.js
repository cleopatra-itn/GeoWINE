import React from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import SelectImage from 'components/SelectImage';
import UploadImage from 'components/UploadImage';

class InputImageTab extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            key: 'selectImage'
        };
    }

    callbackResponse = (newResponseData) => {
        console.log('We got the child response data!!!')
        this.props.appCallback(newResponseData);
    }

    render () {
        return (
            <Tabs
                id="controlled-tab-example"
                activeKey={this.key}
                onSelect={(k) => this.setState({ key: k })}
                >
                <Tab eventKey="selectImageTab" title="Select">
                    <SelectImage inputImageCallback={this.callbackResponse} />
                </Tab>
                <Tab eventKey="uploadImageTab" title="Upload">
                    <UploadImage inputImageCallback={this.callbackResponse} />
                </Tab>
            </Tabs>
        );
    }
}

export default InputImageTab;
