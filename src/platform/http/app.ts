import express from 'express';
import cors from 'cors';

const app = express();

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
    res.json({
        message: 'Hello Sophia3!'
    });
});

// Run API  
//const port = process.env.PORT || 3002;

export default app;

/* app.listen(port, () => {
    console.log(`API rodando em http://localhost:${port}`); 
}); */
