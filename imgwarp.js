// A modernized version of https://github.com/ppisljar/image-morph-js & https://github.com/cxcxcxcx/imgwarp-js
// I have no clue what this code really does, the maths is beyond me, but it seems to work

class AffineDeformation {
    constructor(fromPoints, toPoints, alpha) {
        this.w = null;
        this.pRelative = null;
        this.qRelative = null;
        this.A = null;
        if (fromPoints.length !== toPoints.length) {
            console.error('Points are not of same length.');
            return;
        }
        this.n = fromPoints.length;
        this.fromPoints = fromPoints;
        this.toPoints = toPoints;
        this.alpha = alpha;
    }

    pointMover(point) {
        if (null == this.pRelative || this.pRelative.length < this.n) {
            this.pRelative = new Array(this.n);
        }
        if (null == this.qRelative || this.qRelative.length < this.n) {
            this.qRelative = new Array(this.n);
        }
        if (null == this.w || this.w.length < this.n) {
            this.w = new Array(this.n);
        }
        if (null == this.A || this.A.length < this.n) {
            this.A = new Array(this.n);
        }

        for (let i = 0; i < this.n; ++i) {
            const t = this.fromPoints[i].subtract(point);
            this.w[i] = Math.pow(t.x * t.x + t.y * t.y, -this.alpha);
        }

        const pAverage = Point.weightedAverage(this.fromPoints, this.w);
        const qAverage = Point.weightedAverage(this.toPoints, this.w);

        for (let i = 0; i < this.n; ++i) {
            this.pRelative[i] = this.fromPoints[i].subtract(pAverage);
            this.qRelative[i] = this.toPoints[i].subtract(qAverage);
        }

        const B = new Matrix22(0, 0, 0, 0);
        for (let i = 0; i < this.n; ++i) {
            B.addM(this.pRelative[i].wXtX(this.w[i]));
        }

        const BInv = B.inverse();
        for (let j = 0; j < this.n; ++j) {
            this.A[j] = point.subtract(pAverage).multiply(BInv)
                .dotP(this.pRelative[j]) * this.w[j];
        }

        let r = qAverage; // r is an point
        for (let j = 0; j < this.n; ++j) {
            r = r.add(this.qRelative[j].multiplyD(this.A[j]));
        }
        return r;
    }
}

class BilinearInterpolation {
    constructor(width, height, fillColor) {
        this.width = width;
        this.height = height;
        this.fillColor = fillColor;
        this.imgTargetData = document.createElement('canvas').getContext('2d').createImageData(this.width, this.height);
    }

    generate(source, fromGrid, toGrid) {
        this.imgData = source;
        for (let i = 0; i < toGrid.length; ++i)
            this.fill(toGrid[i], fromGrid[i]);
        return this.imgTargetData;
    }

    fill(sourcePoints, fillingPoints) {
        const x0 = Math.max(fillingPoints[0].x, 0);
        const x1 = Math.min(fillingPoints[2].x, this.width - 1);
        const y0 = Math.max(fillingPoints[0].y, 0);
        const y1 = Math.min(fillingPoints[2].y, this.height - 1);

        let srcX, srcY;
        let xl, xr, topX, topY, bottomX, bottomY;
        let yl, yr, index;

        for (let i = x0; i <= x1; ++i) {
            xl = (i - x0) / (x1 - x0);
            xr = 1 - xl;
            topX = xr * sourcePoints[0].x + xl * sourcePoints[1].x;
            topY = xr * sourcePoints[0].y + xl * sourcePoints[1].y;
            bottomX = xr * sourcePoints[3].x + xl * sourcePoints[2].x;
            bottomY = xr * sourcePoints[3].y + xl * sourcePoints[2].y;

            for (let j = y0; j <= y1; ++j) {
                yl = (j - y0) / (y1 - y0);
                yr = 1 - yl;
                srcX = topX * yr + bottomX * yl;
                srcY = topY * yr + bottomY * yl;
                index = ((j * this.width) + i) * 4;

                if (srcX < 0 || srcX > this.width - 1 ||
                    srcY < 0 || srcY > this.height - 1) {
                    this.imgTargetData.data[index] = this.fillColor[0] || 0;
                    this.imgTargetData.data[index + 1] = this.fillColor[1] || 0;
                    this.imgTargetData.data[index + 2] = this.fillColor[2] || 0;
                    this.imgTargetData.data[index + 3] = this.fillColor[3] || 0;
                    continue;
                }

                const srcX1 = Math.floor(srcX);
                const srcY1 = Math.floor(srcY);
                const base = ((srcY1 * this.width) + srcX1) * 4;
                this.imgTargetData.data[index] = this.imgData[base];
                this.imgTargetData.data[index + 1] = this.imgData[base + 1];
                this.imgTargetData.data[index + 2] = this.imgData[base + 2];
                this.imgTargetData.data[index + 3] = this.imgData[base + 3];
            }
        }
    }
}

class Matrix22 {
    constructor(N11, N12, N21, N22) {
        this.M11 = N11;
        this.M12 = N12;
        this.M21 = N21;
        this.M22 = N22;
    }

    adjugate() {
        return new Matrix22(
            this.M22, -this.M12,
            -this.M21, this.M11);
    }

    determinant() {
        return this.M11 * this.M22 - this.M12 * this.M21;
    }

    multiply(m) {
        this.M11 *= m;
        this.M12 *= m;
        this.M21 *= m;
        this.M22 *= m;
        return this;
    }

    addM(o) {
        this.M11 += o.M11;
        this.M12 += o.M12;
        this.M21 += o.M21;
        this.M22 += o.M22;
    }

    inverse() {
        return this.adjugate().multiply(1.0 / this.determinant());
    }
}

class Point {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }

    add(o) {
        return new Point(this.x + o.x, this.y + o.y);
    }

    subtract(o) {
        return new Point(this.x - o.x, this.y - o.y);
    }

    // w * [x; y] * [x, y]
    wXtX(w) {
        return (new Matrix22(
            this.x * this.x * w, this.x * this.y * w,
            this.y * this.x * w, this.y * this.y * w
        ));
    }

    // Dot product
    dotP(o) {
        return this.x * o.x + this.y * o.y;
    }

    multiply(o) {
        return new Point(
            this.x * o.M11 + this.y * o.M21, this.x * o.M12 + this.y * o.M22);
    }

    multiplyD(o) {
        return new Point(this.x * o, this.y * o);
    }

    static weightedAverage(p, w) {
        let sx = 0,
            sy = 0,
            sw = 0;
        for (let i = 0; i < p.length; i++) {
            sx += p[i].x * w[i];
            sy += p[i].y * w[i];
            sw += w[i];
        }
        return new Point(sx / sw, sy / sw);
    }
}

class Warper {
    constructor(imgData, optGridSize, optAlpha, optFillRGBA) {
        this.alpha = optAlpha || 1;
        this.gridSize = optGridSize || 20;
        this.fillColor = optFillRGBA || [0, 0, 0, 0];

        this.width = imgData.width;
        this.height = imgData.height;
        this.imgData = imgData.data;
        this.bilinearInterpolation = new BilinearInterpolation(this.width, this.height, this.fillColor);

        this.grid = [];
        for (let i = 0; i < this.width ; i += this.gridSize)
            for (let j = 0; j < this.height ; j += this.gridSize)
                this.grid.push([
                    new Point(i,j),
                    new Point(i + this.gridSize, j),
                    new Point(i + this.gridSize, j + this.gridSize),
                    new Point(i, j + this.gridSize),
                ]);
    }

    warp(fromPoints, toPoints) {
        const deformation = new AffineDeformation(toPoints, fromPoints, this.alpha);
        const transformedGrid = [];
        for (let i = 0; i < this.grid.length; ++i)
            transformedGrid.push([
                deformation.pointMover(this.grid[i][0]),
                deformation.pointMover(this.grid[i][1]),
                deformation.pointMover(this.grid[i][2]),
                deformation.pointMover(this.grid[i][3]),
            ]);

        return this.bilinearInterpolation.generate(this.imgData, this.grid, transformedGrid);
    };
}

module.exports = {
    Point,
    Warper,
};
