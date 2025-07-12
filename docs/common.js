const scripts = [ ...document.querySelectorAll('textarea') ];
let queue = Promise.resolve();

scripts.forEach( ( script ) => {
    const outputwrapper = document.createElement('div');
    const div = document.createElement('details');
    div.open = true;
    const button = document.createElement('button');
    button.innerText = 'Run';
    const pre = document.createElement('textarea');
    const iInput = document.createElement('input');
    const float = document.createElement('summary');
    float.tabIndex = -1;
    iInput.type = 'number';
    iInput.value = script.dataset.iterations;

    div.onkeydown = ( event ) => {
        if ( event.key === 'Enter' && event.shiftKey ) {
            event.preventDefault();
            button.click();
        }
    };

    function stringifyArray( array ) {
        array = Array.from( array );
        // Only show first 3 and last 3 if larger than 6.
        if ( array.length > 6 ) {
            return `[ ${array.slice(0,3).join(', ')}, ..., ${array.slice(-3).join(', ')}]`;
        }
        return `[ ${array.join(', ')} ]`;
    }

    function stringify( data ) {
        if ( ( window.FloatMatrix && data instanceof FloatMatrix ) || ( window.Int32Array && data instanceof Int32Array ) ) {
            if ( data.shape.length === 1 ) return `${data.constructor.name}(${data.length}) ${ stringifyArray( data ) }`;

            // If larger than 6 rows, get the first 3 and last 3.
            if (data.shape.length === 3) {
                const [depth, height, width] = data.shape;
                const slices = [];
                for (let d = 0; d < (depth > 6 ? 3 : depth); d++) {
                    const rows = [];
                    for (let h = 0; h < (height > 6 ? 3 : height); h++) {
                        const row = [];
                        for (let w = 0; w < width; w++) {
                            row.push(data[d * height * width + h * width + w]);
                        }
                        rows.push(stringifyArray(row));
                    }
                    if (height > 6) {
                        rows.push('...');
                        for (let h = height - 3; h < height; h++) {
                            const row = [];
                            for (let w = 0; w < width; w++) {
                                row.push(data[d * height * width + h * width + w]);
                            }
                            rows.push(stringifyArray(row));
                        }
                    }
                    slices.push(`[\n  ${rows.join(',\n  ')}\n ]`);
                }
                if (depth > 6) {
                    slices.push('...');
                    for (let d = depth - 3; d < depth; d++) {
                        const rows = [];
                        for (let h = 0; h < (height > 6 ? 3 : height); h++) {
                            const row = [];
                            for (let w = 0; w < width; w++) {
                                row.push(data[d * height * width + h * width + w]);
                            }
                            rows.push(stringifyArray(row));
                        }
                        if (height > 6) {
                            rows.push('...');
                            for (let h = height - 3; h < height; h++) {
                                const row = [];
                                for (let w = 0; w < width; w++) {
                                    row.push(data[d * height * width + h * width + w]);
                                }
                                rows.push(stringifyArray(row));
                            }
                        }
                        slices.push(`[\n  ${rows.join(',\n  ')}\n ]`);
                    }
                }
                return `${data.shape.join('×')} [\n${slices.join(',\n')}\n]`;
            } else if (data.shape.length === 2) {
                if (data.shape[0] > 6) {
                    const rows = [];
                    for (let m = 0; m < 3; m++) {
                        const row = [];
                        for (let n = 0; n < data.shape[1]; n++) {
                            row.push(data[m * data.shape[1] + n]);
                        }
                        rows.push(stringifyArray(row));
                    }
                    rows.push('...');
                    for (let m = data.shape[0] - 3; m < data.shape[0]; m++) {
                        const row = [];
                        for (let n = 0; n < data.shape[1]; n++) {
                            row.push(data[m * data.shape[1] + n]);
                        }
                        rows.push(stringifyArray(row));
                    }
                    return `${data.shape.join('×')} [
${rows.join(',\n ')}
]`;
                }

                const rows = [];
                for (let m = 0; m < data.shape[0]; m++) {
                    const row = [];
                    for (let n = 0; n < data.shape[1]; n++) {
                        row.push(data[m * data.shape[1] + n]);
                    }
                    rows.push(stringifyArray(row));
                }
                return `${data.shape.join('×')} [
${rows.join(',\n ')}
]`;
            }
        }

        function hellip( string, condition ) {
            return condition ? `${string.slice(0,-1)}…` : string;
        }

        if ( typeof data === 'string' ) return hellip( JSON.stringify( data.slice( 0, 100 ) ), data.length > 100 );
        if ( typeof data === 'number' ) return data.toString();
        if ( typeof data === 'boolean' ) return data.toString();
        if ( typeof data === 'undefined' ) return 'undefined';
        if ( data === null ) return 'null';
        if ( data instanceof Error ) return data.toString();
        if ( data instanceof Array || data instanceof Float32Array || data instanceof Int32Array ) {
            return `${ data.constructor.name }(${data.length}) ${ stringifyArray( data ) }`;
        }
        if ( data instanceof Set ) {
            return `Set(${data.size}) ${ stringifyArray( [...data] ) }`;
        }
        if ( typeof data === 'object' ) return JSON.stringify( data, ( key, value ) => {
            if ( ! key ) return value;
            if ( typeof value === 'function' ) return '[Function]';
            if ( typeof value === 'object' ) return '[Object]';
            return value;
        }, 1 ).replace( /\n\s*/g, ' ' );
        if ( typeof data === 'function' ) return `Function`;
    }

    button.tabIndex = -1;
    button.onclick = async () => {
        button.disabled = true;
        outputwrapper.innerHTML = '';
        const output = document.createElement('pre');
        outputwrapper.append( output );
        outputwrapper.focus();
        pre?.editor.save();
        let text = pre.value;

        const ast = acorn.parse(text, { ecmaVersion: 'latest', sourceType: 'module' });
        console.log(ast);

        // collect all top-level declarations names.
        const declarations = [];
        const replacements = [];
        for ( const dt of ast.body ) {
            if ( dt.type === 'VariableDeclaration' ) {
                for ( const decl of dt.declarations ) {
                    switch ( decl.id.type ) {
                        case 'Identifier':
                            declarations.push( decl.id.name );
                            break;
                        case 'ObjectPattern':
                            for ( const prop of decl.id.properties ) {
                                declarations.push( prop.key.name );
                            }
                            break;
                        case 'ArrayPattern':
                            for ( const elem of decl.id.elements ) {
                                declarations.push( elem.name );
                            }
                            break;
                    }
                }
            } else if ( dt.type === 'FunctionDeclaration' ) {
                declarations.push( dt.id.name );
            } else if ( dt.type === 'ClassDeclaration' ) {
                declarations.push( dt.id.name );
            } else if ( dt.type === 'ImportDeclaration' ) {
                if ( dt.source.value.startsWith( './' ) ) {
                    replacements.push( {
                        start: dt.source.start,
                        end: dt.source.end,
                        replacement: JSON.stringify( new URL( dt.source.value, location ).href ),
                    } );
                }

                for ( const specifier of dt.specifiers ) {
                    declarations.push( specifier.local.name );
                }
            }
        }

        for (const { start, end, replacement } of replacements.reverse()) {
            text = text.slice(0, start) + replacement + text.slice(end);
        }

        console.log(text);

        text += `;${declarations.map( decl =>
            `window.${decl} = ${decl};print( ${decl}, '${decl}' );`
        ).join( '\n' )}`;

        const blob = new Blob( [ text ], { type: 'text/javascript' } );

        let i = parseInt( iInput.value, 10 ) || 1;

        const promiseExecutor = async (resolve, reject) => {
            const url = URL.createObjectURL(blob);
            print = function ( data, key = '' ) {
                const line = document.createElement('div');
                console.log(data);
                if ( data instanceof Element ) {
                    if (!output.contains(data)) {
                        line.appendChild( data );
                    }
                } else if ( Array.isArray( data ) && data.every( child => child instanceof Element ) ) {
                    line.style.display = 'flex';
                    data.forEach( child => line.appendChild( child ) );
                } else {
                    if ( key ) {
                        const b = document.createElement('b');
                        b.textContent = key;
                        line.appendChild( b );
                    }
                    line.appendChild(
                        document.createTextNode( ( key ? ': ' : '' ) + stringify( data ) )
                    );
                }
                output.appendChild( line );
            }
            try {
                const imports = await import(url);
                Object.keys(imports).forEach((key) => {
                    window[key] = imports[key];
                    print(imports[key], key);
                });
            } catch (error) {
                output.dataset.error = true;
                print(error);
            }

            resolve();
        };

        queue = queue.then( () => new Promise( promiseExecutor ) ).then( () => {
            button.disabled = false;
        } );
    };

    div.onfocus = () => {
        div.open = true;
    };

    pre.button = button;
    pre.style.width = '100%';
    pre.value = script.value.trim();
    pre.rows = pre.value.split( '\n' ).length;
    iInput.style.width = '4em';
    if ( script.dataset.src ) {
        const code = document.createElement('code');
        code.textContent = script.dataset.src;
        float.appendChild( code );
        float.appendChild( document.createTextNode( ' ' ) );
    }
    float.appendChild( button );
    if ( script.dataset.iterations !== undefined ) {
        float.appendChild( document.createTextNode( ' × ' ) );
        float.appendChild( iInput );
    }
    div.appendChild( float );
    div.appendChild( pre );
    div.id = script.id;
    script.replaceWith( div );
    div.after( outputwrapper );
} );

const article = document.querySelector('article');

[...article.children].forEach( ( block ) => {
    block.tabIndex = 0;
    block.setAttribute( 'aria-label', 'Shift+Enter to continue' );
} );

article.addEventListener('keydown', ( event ) => {
    if ( event.key === 'Enter' && event.shiftKey && ! event.defaultPrevented ) {
        document.activeElement.closest('[aria-label]').nextElementSibling?.focus();
    }
})

article.firstElementChild.focus();
