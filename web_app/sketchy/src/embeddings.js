import React, { useState, useCallback, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Plotly, Plot } from 'react-plotly.js'
import { Box, ChakraProvider, Button, Text, Heading } from '@chakra-ui/react'

const darkGray = "#A3A8B0"
const textColor = "#FFFFFF"
const backgroundColor = "#1A365D"





function Embeddings() {
    const [x, setX] = useState([])
    const [y, setY] = useState([])
    const [z, setZ] = useState([])
    const [legend, setLegend] = useState([])


    async function initialiseData() {
        // Send to back end
        const response = await fetch('/get_embeddings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })

        // Receive response
        if (response.ok) {
            const res = await response.json()
            let x = res["x"]
            let y = res["y"]

            setX(res["x"])
            setY(res["y"])

        }
    }


    // function makeplot() {
    //     console.log("make plot")
    //     Plotly.d3.csv("embeddings_pca.csv", function (data) {
    //         console.log("make plot inside")
    //         processData(data)
    //     })
    // }

    // function processData(allRows) {
    //     console.log("process data")
    //     let x = [], y = [], z = [], legend = [];
    //     for (var i = 0; i < allRows.length; i++) {
    //         let row = allRows[i];
    //         x.push(row['embeddings_1']);
    //         y.push(row['embeddings_2']);
    //         z.push(row['embeddings_3']);
    //         legend.push(row['classes']);
    //     }
    //     console.log('X', x, 'Y', y, 'z', z, 'legend', legend)
    //     setX(x)
    //     setY(y)
    //     setZ(z)
    //     setLegend(legend)
    // }

    // makeplot()

    return (
        <ChakraProvider >
            <Box bg={backgroundColor}>
                <Heading fontSize="4xl" color={textColor} align="center">
                    AMLD 2021 Visium's Sketchy App
                </Heading>
                <Text fontSize="xs" color={textColor} align="center">
                    --------------------------------------------------------
                </Text>
                <Text fontSize="xl" color={textColor} align="center">
                    Embeddings: Images and Sketches in latent space
                </Text>


                <form class="form-inline">
                    <div class="form-group">
                        <label for="files">Upload a CSV formatted file:</label>
                        <input type="file" id="files" class="form-control" accept=".csv" required />
                    </div>
                    <div class="form-group">
                        <button type="submit" id="submit-file" class="btn btn-primary">Upload File</button>
                    </div>
                </form>
                {/* 
                <Plot
                    data={[
                        {
                            x: { x },
                            y: { y },
                            z: { z },
                            mode: 'markers',
                            type: 'scatter3d',
                            marker: {
                                size: 8,
                                opacity: 0.8,
                                color: 'rgba(100, 100, 217, 1)'
                            },
                        }
                    ]}
                    layout={{
                        height: 800,
                        width: 1200,
                        title: `3D Views`,
                    }}
                    onRelayout={(figure) => console.log(figure)}
                /> */}

                <Link to="/drawing" className="embeddings_link">
                    <Button color={backgroundColor} border="2px" borderColor={darkGray} variant="solid" size="lg"> Back to Drawing</Button>
                </Link>
            </Box>
        </ChakraProvider >
    )
}

export default Embeddings