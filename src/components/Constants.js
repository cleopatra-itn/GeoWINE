import React from 'react';
import { components } from 'react-select';

export const IMAGES = [
    {
        original: require('images/input/Q2981.jpeg').default,
        thumbnail: require('images/input/Q2981.jpeg').default,
        id: 'Q2981.jpeg'
    },
    {
        original: require('images/input/Q11259.jpeg').default,
        thumbnail: require('images/input/Q11259.jpeg').default,
        id: 'Q11259.jpeg'
    },
    {
        original: require('images/input/Q20410627.jpg').default,
        thumbnail: require('images/input/Q20410627.jpg').default,
        id: 'Q20410627.jpg'
    },
    {
        original: require('images/input/Q9141.jpeg').default,
        thumbnail: require('images/input/Q9141.jpeg').default,
        id: 'Q9141.jpeg'
    },
    {
        original: require('images/input/Q36332.jpg').default,
        thumbnail: require('images/input/Q36332.jpg').default,
        id: 'Q36332.jpg'
    },
    {
        original: require('images/input/Q4994352.jpg').default,
        thumbnail: require('images/input/Q4994352.jpg').default,
        id: 'Q4994352.jpg'
    },
    {
        original: require('images/input/Q1045876.jpg').default,
        thumbnail: require('images/input/Q1045876.jpg').default,
        id: 'Q1045876.jpg'
    },
    {
        original: require('images/input/Q676203.jpg').default,
        thumbnail: require('images/input/Q676203.jpg').default,
        id: 'Q676203.jpg'
    },
    {
        original: require('images/input/Q2630165.jpg').default,
        thumbnail: require('images/input/Q2630165.jpg').default,
        id: 'Q2630165.jpg'
    },
    {
        original: require('images/input/Q9202.jpg').default,
        thumbnail: require('images/input/Q9202.jpg').default,
        id: 'Q9202.jpg'
    },
    {
        original: require('images/input/Q5788.jpg').default,
        thumbnail: require('images/input/Q5788.jpg').default,
        id: 'Q5788.jpg'
    },
    {
        original: require('images/input/Q12495.jpg').default,
        thumbnail: require('images/input/Q12495.jpg').default,
        id: 'Q12495.jpg'
    },
    {
        original: require('images/input/Q45178.jpg').default,
        thumbnail: require('images/input/Q45178.jpg').default,
        id: 'Q45178.jpg'
    },
    {
        original: require('images/input/Q79961.jpg').default,
        thumbnail: require('images/input/Q79961.jpg').default,
        id: 'Q79961.jpg'
    },
    {
        original: require('images/input/Q133274.jpg').default,
        thumbnail: require('images/input/Q133274.jpg').default,
        id: 'Q133274.jpg'
    },
    {
        original: require('images/input/Q134883.jpg').default,
        thumbnail: require('images/input/Q134883.jpg').default,
        id: 'Q134883.jpg'
    },
    {
        original: require('images/input/Q13217298.jpg').default,
        thumbnail: require('images/input/Q13217298.jpg').default,
        id: 'Q13217298.jpg'
    },
];

export const TYPES = [
    { value: 'tourist attraction', label: 'Tourist Attraction' },
    { value: 'religious building', label: 'Religious Building' },
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

export const ValueContainer = ({ children, getValue, ...props }) => {
    var length = getValue().length;

    return (
      <components.ValueContainer {...props}>
        {!props.selectProps.menuIsOpen &&
          `${length} selected`}
        {React.cloneElement(children[1])}
      </components.ValueContainer>
    );
  };